; child-snack task with 3 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 681389

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child463 child93 child222 - child
    bread217 bread279 bread499 - bread-portion
    content183 content415 content363 - content-portion
    tray451 tray248 tray475 tray271 - tray
    table115 table96 table398 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray451 kitchen)
     (at tray248 kitchen)
     (at tray475 kitchen)
     (at tray271 kitchen)
     (at_kitchen_bread bread217)
     (at_kitchen_bread bread279)
     (at_kitchen_bread bread499)
     (at_kitchen_content content183)
     (at_kitchen_content content415)
     (at_kitchen_content content363)
     (no_gluten_bread bread499)
     (no_gluten_bread bread279)
     (no_gluten_content content363)
     (no_gluten_content content183)
     (allergic_gluten child463)
     (allergic_gluten child222)
     (not_allergic_gluten child93)
     (waiting child463 table398)
     (waiting child93 table96)
     (waiting child222 table96)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child463)
     (served child93)
     (served child222)
    )
  )
)
