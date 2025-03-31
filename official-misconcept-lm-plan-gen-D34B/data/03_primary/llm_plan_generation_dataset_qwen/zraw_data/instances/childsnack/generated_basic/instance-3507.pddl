; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 495400

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child390 child389 - child
    bread3 bread304 - bread-portion
    content65 content368 - content-portion
    tray443 tray93 - tray
    table314 table433 table179 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray443 kitchen)
     (at tray93 kitchen)
     (at_kitchen_bread bread3)
     (at_kitchen_bread bread304)
     (at_kitchen_content content65)
     (at_kitchen_content content368)
     (no_gluten_bread bread304)
     (no_gluten_content content368)
     (allergic_gluten child389)
     (not_allergic_gluten child390)
     (waiting child390 table433)
     (waiting child389 table179)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child390)
     (served child389)
    )
  )
)
