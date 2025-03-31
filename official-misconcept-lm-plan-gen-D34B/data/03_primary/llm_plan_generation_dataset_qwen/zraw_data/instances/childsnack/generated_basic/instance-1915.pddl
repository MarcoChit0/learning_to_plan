; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 511085

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child417 - child
    bread137 - bread-portion
    content134 - content-portion
    tray492 tray348 tray92 - tray
    table124 table426 table207 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray492 kitchen)
     (at tray348 kitchen)
     (at tray92 kitchen)
     (at_kitchen_bread bread137)
     (at_kitchen_content content134)
     (not_allergic_gluten child417)
     (waiting child417 table124)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child417)
    )
  )
)
