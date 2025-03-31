; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 915841

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child372 - child
    bread191 - bread-portion
    content277 - content-portion
    tray397 tray383 tray341 tray450 - tray
    table414 table335 table0 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray397 kitchen)
     (at tray383 kitchen)
     (at tray341 kitchen)
     (at tray450 kitchen)
     (at_kitchen_bread bread191)
     (at_kitchen_content content277)
     (not_allergic_gluten child372)
     (waiting child372 table335)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child372)
    )
  )
)
