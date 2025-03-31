; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 777190

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child455 - child
    bread207 - bread-portion
    content219 - content-portion
    tray481 - tray
    table450 table233 table422 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray481 kitchen)
     (at_kitchen_bread bread207)
     (at_kitchen_content content219)
     (not_allergic_gluten child455)
     (waiting child455 table450)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child455)
    )
  )
)
