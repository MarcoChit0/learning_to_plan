; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 232724

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child434 child315 - child
    bread6 bread227 - bread-portion
    content368 content394 - content-portion
    tray362 tray461 - tray
    table229 table287 table50 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray362 kitchen)
     (at tray461 kitchen)
     (at_kitchen_bread bread6)
     (at_kitchen_bread bread227)
     (at_kitchen_content content368)
     (at_kitchen_content content394)
     (not_allergic_gluten child434)
     (not_allergic_gluten child315)
     (waiting child434 table287)
     (waiting child315 table287)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child434)
     (served child315)
    )
  )
)
