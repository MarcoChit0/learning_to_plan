; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 246300

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child492 - child
    bread279 - bread-portion
    content422 - content-portion
    tray288 tray16 tray222 tray291 - tray
    table306 table230 table146 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray288 kitchen)
     (at tray16 kitchen)
     (at tray222 kitchen)
     (at tray291 kitchen)
     (at_kitchen_bread bread279)
     (at_kitchen_content content422)
     (not_allergic_gluten child492)
     (waiting child492 table306)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child492)
    )
  )
)
