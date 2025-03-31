; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 740728

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child194 - child
    bread50 - bread-portion
    content365 - content-portion
    tray25 - tray
    table28 table25 table368 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray25 kitchen)
     (at_kitchen_bread bread50)
     (at_kitchen_content content365)
     (not_allergic_gluten child194)
     (waiting child194 table25)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child194)
    )
  )
)
