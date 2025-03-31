; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 596917

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child424 child257 - child
    bread273 bread170 - bread-portion
    content492 content404 - content-portion
    tray237 - tray
    table303 table311 table217 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray237 kitchen)
     (at_kitchen_bread bread273)
     (at_kitchen_bread bread170)
     (at_kitchen_content content492)
     (at_kitchen_content content404)
     (no_gluten_bread bread170)
     (no_gluten_content content492)
     (allergic_gluten child424)
     (not_allergic_gluten child257)
     (waiting child424 table217)
     (waiting child257 table217)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child424)
     (served child257)
    )
  )
)
