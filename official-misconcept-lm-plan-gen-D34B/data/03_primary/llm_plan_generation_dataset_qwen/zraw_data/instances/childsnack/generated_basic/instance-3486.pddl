; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 570512

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child436 child355 - child
    bread182 bread120 - bread-portion
    content36 content70 - content-portion
    tray144 tray130 - tray
    table34 table88 table397 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray144 kitchen)
     (at tray130 kitchen)
     (at_kitchen_bread bread182)
     (at_kitchen_bread bread120)
     (at_kitchen_content content36)
     (at_kitchen_content content70)
     (no_gluten_bread bread120)
     (no_gluten_content content70)
     (allergic_gluten child436)
     (not_allergic_gluten child355)
     (waiting child436 table34)
     (waiting child355 table397)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child436)
     (served child355)
    )
  )
)
