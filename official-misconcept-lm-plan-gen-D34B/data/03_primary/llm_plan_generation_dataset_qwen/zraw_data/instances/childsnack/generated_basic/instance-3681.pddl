; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 688914

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child233 child368 - child
    bread368 bread385 - bread-portion
    content90 content466 - content-portion
    tray169 tray395 - tray
    table457 table273 table362 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray169 kitchen)
     (at tray395 kitchen)
     (at_kitchen_bread bread368)
     (at_kitchen_bread bread385)
     (at_kitchen_content content90)
     (at_kitchen_content content466)
     (no_gluten_bread bread385)
     (no_gluten_content content466)
     (allergic_gluten child368)
     (not_allergic_gluten child233)
     (waiting child233 table362)
     (waiting child368 table362)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child233)
     (served child368)
    )
  )
)
