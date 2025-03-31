; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 615023

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child422 - child
    bread243 - bread-portion
    content185 - content-portion
    tray260 tray288 tray363 tray119 - tray
    table278 table368 table237 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray260 kitchen)
     (at tray288 kitchen)
     (at tray363 kitchen)
     (at tray119 kitchen)
     (at_kitchen_bread bread243)
     (at_kitchen_content content185)
     (not_allergic_gluten child422)
     (waiting child422 table278)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child422)
    )
  )
)
