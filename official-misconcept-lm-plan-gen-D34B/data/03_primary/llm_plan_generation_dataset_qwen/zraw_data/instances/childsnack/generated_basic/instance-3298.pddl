; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 147221

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child443 child284 - child
    bread352 bread21 - bread-portion
    content410 content371 - content-portion
    tray277 tray473 tray68 tray117 - tray
    table406 table147 table66 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray277 kitchen)
     (at tray473 kitchen)
     (at tray68 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread352)
     (at_kitchen_bread bread21)
     (at_kitchen_content content410)
     (at_kitchen_content content371)
     (no_gluten_bread bread21)
     (no_gluten_content content410)
     (allergic_gluten child284)
     (not_allergic_gluten child443)
     (waiting child443 table147)
     (waiting child284 table406)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child443)
     (served child284)
    )
  )
)
