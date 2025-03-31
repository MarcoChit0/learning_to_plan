; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 990349

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child167 child367 - child
    bread136 bread269 - bread-portion
    content473 content368 - content-portion
    tray449 tray164 - tray
    table172 table320 table386 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray449 kitchen)
     (at tray164 kitchen)
     (at_kitchen_bread bread136)
     (at_kitchen_bread bread269)
     (at_kitchen_content content473)
     (at_kitchen_content content368)
     (not_allergic_gluten child167)
     (not_allergic_gluten child367)
     (waiting child167 table172)
     (waiting child367 table386)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child167)
     (served child367)
    )
  )
)
