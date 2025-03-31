; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 529745

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child352 child157 - child
    bread192 bread316 - bread-portion
    content229 content423 - content-portion
    tray174 tray305 tray140 tray186 - tray
    table203 table354 table172 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray174 kitchen)
     (at tray305 kitchen)
     (at tray140 kitchen)
     (at tray186 kitchen)
     (at_kitchen_bread bread192)
     (at_kitchen_bread bread316)
     (at_kitchen_content content229)
     (at_kitchen_content content423)
     (no_gluten_bread bread192)
     (no_gluten_content content423)
     (allergic_gluten child157)
     (not_allergic_gluten child352)
     (waiting child352 table203)
     (waiting child157 table354)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child352)
     (served child157)
    )
  )
)
