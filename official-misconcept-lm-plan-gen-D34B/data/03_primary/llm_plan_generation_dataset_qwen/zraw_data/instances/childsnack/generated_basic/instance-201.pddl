; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 696216

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child362 child276 - child
    bread237 bread481 - bread-portion
    content182 content351 - content-portion
    tray4 tray36 tray251 - tray
    table333 table168 table385 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray4 kitchen)
     (at tray36 kitchen)
     (at tray251 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_bread bread481)
     (at_kitchen_content content182)
     (at_kitchen_content content351)
     (not_allergic_gluten child362)
     (not_allergic_gluten child276)
     (waiting child362 table333)
     (waiting child276 table168)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child362)
     (served child276)
    )
  )
)
