; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 369815

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child247 child338 - child
    bread386 bread362 - bread-portion
    content180 content312 - content-portion
    tray225 tray290 tray122 tray145 - tray
    table406 table393 table418 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray225 kitchen)
     (at tray290 kitchen)
     (at tray122 kitchen)
     (at tray145 kitchen)
     (at_kitchen_bread bread386)
     (at_kitchen_bread bread362)
     (at_kitchen_content content180)
     (at_kitchen_content content312)
     (not_allergic_gluten child247)
     (not_allergic_gluten child338)
     (waiting child247 table406)
     (waiting child338 table393)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child247)
     (served child338)
    )
  )
)
