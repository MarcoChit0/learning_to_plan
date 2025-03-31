; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 463053

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child220 child359 - child
    bread438 bread170 - bread-portion
    content439 content222 - content-portion
    tray94 tray282 - tray
    table405 table40 table35 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray94 kitchen)
     (at tray282 kitchen)
     (at_kitchen_bread bread438)
     (at_kitchen_bread bread170)
     (at_kitchen_content content439)
     (at_kitchen_content content222)
     (no_gluten_bread bread438)
     (no_gluten_content content222)
     (allergic_gluten child359)
     (not_allergic_gluten child220)
     (waiting child220 table40)
     (waiting child359 table35)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child220)
     (served child359)
    )
  )
)
