; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 722914

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child55 - child
    bread407 - bread-portion
    content327 - content-portion
    tray302 tray473 tray402 - tray
    table203 table488 table492 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at tray473 kitchen)
     (at tray402 kitchen)
     (at_kitchen_bread bread407)
     (at_kitchen_content content327)
     (not_allergic_gluten child55)
     (waiting child55 table492)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child55)
    )
  )
)
