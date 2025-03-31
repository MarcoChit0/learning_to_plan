; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 359009

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child237 child380 child438 - child
    bread52 bread262 bread139 - bread-portion
    content283 content362 content297 - content-portion
    tray303 tray326 tray130 - tray
    table187 table494 table130 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray303 kitchen)
     (at tray326 kitchen)
     (at tray130 kitchen)
     (at_kitchen_bread bread52)
     (at_kitchen_bread bread262)
     (at_kitchen_bread bread139)
     (at_kitchen_content content283)
     (at_kitchen_content content362)
     (at_kitchen_content content297)
     (not_allergic_gluten child237)
     (not_allergic_gluten child438)
     (not_allergic_gluten child380)
     (waiting child237 table187)
     (waiting child380 table130)
     (waiting child438 table187)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child237)
     (served child380)
     (served child438)
    )
  )
)
