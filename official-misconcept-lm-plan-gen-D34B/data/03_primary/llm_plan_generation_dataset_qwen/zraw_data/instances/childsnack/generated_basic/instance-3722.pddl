; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 318384

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child59 - child
    bread462 - bread-portion
    content102 - content-portion
    tray256 tray53 tray165 - tray
    table140 table206 table371 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray256 kitchen)
     (at tray53 kitchen)
     (at tray165 kitchen)
     (at_kitchen_bread bread462)
     (at_kitchen_content content102)
     (not_allergic_gluten child59)
     (waiting child59 table140)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child59)
    )
  )
)
