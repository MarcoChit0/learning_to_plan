; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 15601

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child157 - child
    bread494 - bread-portion
    content468 - content-portion
    tray238 - tray
    table148 table384 table456 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray238 kitchen)
     (at_kitchen_bread bread494)
     (at_kitchen_content content468)
     (not_allergic_gluten child157)
     (waiting child157 table148)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child157)
    )
  )
)
