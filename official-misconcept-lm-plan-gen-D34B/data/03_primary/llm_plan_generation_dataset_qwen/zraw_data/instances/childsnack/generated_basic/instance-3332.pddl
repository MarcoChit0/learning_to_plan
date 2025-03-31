; child-snack task with 2 children and 0.71 gluten factor 
; constant factor of 1.7
; random seed: 423669

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child247 child407 - child
    bread99 bread87 - bread-portion
    content27 content134 - content-portion
    tray371 tray421 tray231 tray47 - tray
    table218 table365 table430 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray371 kitchen)
     (at tray421 kitchen)
     (at tray231 kitchen)
     (at tray47 kitchen)
     (at_kitchen_bread bread99)
     (at_kitchen_bread bread87)
     (at_kitchen_content content27)
     (at_kitchen_content content134)
     (no_gluten_bread bread87)
     (no_gluten_content content134)
     (allergic_gluten child407)
     (not_allergic_gluten child247)
     (waiting child247 table365)
     (waiting child407 table430)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child247)
     (served child407)
    )
  )
)
