; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 586070

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child460 child122 - child
    bread193 bread323 - bread-portion
    content105 content136 - content-portion
    tray128 tray472 tray6 tray203 - tray
    table189 table495 table159 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray128 kitchen)
     (at tray472 kitchen)
     (at tray6 kitchen)
     (at tray203 kitchen)
     (at_kitchen_bread bread193)
     (at_kitchen_bread bread323)
     (at_kitchen_content content105)
     (at_kitchen_content content136)
     (no_gluten_bread bread193)
     (no_gluten_content content105)
     (allergic_gluten child460)
     (not_allergic_gluten child122)
     (waiting child460 table159)
     (waiting child122 table495)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child460)
     (served child122)
    )
  )
)
