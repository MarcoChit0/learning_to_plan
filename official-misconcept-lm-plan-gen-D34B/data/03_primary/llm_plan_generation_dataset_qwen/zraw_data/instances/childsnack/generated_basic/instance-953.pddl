; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 552202

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child487 child220 child424 - child
    bread166 bread283 bread106 - bread-portion
    content383 content317 content359 - content-portion
    tray492 tray302 tray421 tray45 - tray
    table226 table455 table376 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray492 kitchen)
     (at tray302 kitchen)
     (at tray421 kitchen)
     (at tray45 kitchen)
     (at_kitchen_bread bread166)
     (at_kitchen_bread bread283)
     (at_kitchen_bread bread106)
     (at_kitchen_content content383)
     (at_kitchen_content content317)
     (at_kitchen_content content359)
     (not_allergic_gluten child487)
     (not_allergic_gluten child220)
     (not_allergic_gluten child424)
     (waiting child487 table376)
     (waiting child220 table376)
     (waiting child424 table376)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child487)
     (served child220)
     (served child424)
    )
  )
)
