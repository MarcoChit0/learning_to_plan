; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 723867

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child62 - child
    bread172 - bread-portion
    content244 - content-portion
    tray271 tray122 - tray
    table385 table174 table379 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray271 kitchen)
     (at tray122 kitchen)
     (at_kitchen_bread bread172)
     (at_kitchen_content content244)
     (not_allergic_gluten child62)
     (waiting child62 table379)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child62)
    )
  )
)
