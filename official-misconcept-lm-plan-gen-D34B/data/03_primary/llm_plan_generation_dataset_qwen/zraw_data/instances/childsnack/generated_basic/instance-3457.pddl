; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 543990

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child21 - child
    bread390 - bread-portion
    content284 - content-portion
    tray304 tray278 tray53 tray276 - tray
    table155 table440 table414 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray304 kitchen)
     (at tray278 kitchen)
     (at tray53 kitchen)
     (at tray276 kitchen)
     (at_kitchen_bread bread390)
     (at_kitchen_content content284)
     (not_allergic_gluten child21)
     (waiting child21 table414)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child21)
    )
  )
)
