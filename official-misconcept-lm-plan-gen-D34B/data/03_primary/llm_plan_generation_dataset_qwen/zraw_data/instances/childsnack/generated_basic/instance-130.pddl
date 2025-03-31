; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 679776

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child493 child173 - child
    bread439 bread307 - bread-portion
    content93 content160 - content-portion
    tray244 tray145 tray47 tray451 - tray
    table450 table23 table478 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray244 kitchen)
     (at tray145 kitchen)
     (at tray47 kitchen)
     (at tray451 kitchen)
     (at_kitchen_bread bread439)
     (at_kitchen_bread bread307)
     (at_kitchen_content content93)
     (at_kitchen_content content160)
     (not_allergic_gluten child493)
     (not_allergic_gluten child173)
     (waiting child493 table23)
     (waiting child173 table23)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child493)
     (served child173)
    )
  )
)
