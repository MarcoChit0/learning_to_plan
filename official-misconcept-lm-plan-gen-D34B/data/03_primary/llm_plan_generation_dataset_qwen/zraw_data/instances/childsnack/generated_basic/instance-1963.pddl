; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 868230

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child379 child104 - child
    bread129 bread270 - bread-portion
    content440 content368 - content-portion
    tray284 - tray
    table311 table149 table417 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray284 kitchen)
     (at_kitchen_bread bread129)
     (at_kitchen_bread bread270)
     (at_kitchen_content content440)
     (at_kitchen_content content368)
     (no_gluten_bread bread270)
     (no_gluten_content content368)
     (allergic_gluten child379)
     (not_allergic_gluten child104)
     (waiting child379 table149)
     (waiting child104 table417)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child379)
     (served child104)
    )
  )
)
