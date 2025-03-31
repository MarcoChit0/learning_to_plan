; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 251493

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child244 - child
    bread105 - bread-portion
    content67 - content-portion
    tray496 - tray
    table107 table243 table338 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray496 kitchen)
     (at_kitchen_bread bread105)
     (at_kitchen_content content67)
     (not_allergic_gluten child244)
     (waiting child244 table107)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child244)
    )
  )
)
