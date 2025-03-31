; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 595450

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child362 child25 - child
    bread466 bread489 - bread-portion
    content397 content460 - content-portion
    tray241 tray129 tray208 tray309 - tray
    table430 table406 table428 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray241 kitchen)
     (at tray129 kitchen)
     (at tray208 kitchen)
     (at tray309 kitchen)
     (at_kitchen_bread bread466)
     (at_kitchen_bread bread489)
     (at_kitchen_content content397)
     (at_kitchen_content content460)
     (not_allergic_gluten child25)
     (not_allergic_gluten child362)
     (waiting child362 table406)
     (waiting child25 table430)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child362)
     (served child25)
    )
  )
)
