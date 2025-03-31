; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 246014

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child241 child444 - child
    bread226 bread129 - bread-portion
    content283 content89 - content-portion
    tray295 - tray
    table433 table428 table212 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray295 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_bread bread129)
     (at_kitchen_content content283)
     (at_kitchen_content content89)
     (no_gluten_bread bread226)
     (no_gluten_content content283)
     (allergic_gluten child444)
     (not_allergic_gluten child241)
     (waiting child241 table212)
     (waiting child444 table433)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child241)
     (served child444)
    )
  )
)
