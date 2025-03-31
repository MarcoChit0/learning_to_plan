; child-snack task with 3 children and 0.38 gluten factor 
; constant factor of 1.5
; random seed: 14678

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child315 child50 child118 - child
    bread288 bread173 bread193 - bread-portion
    content54 content66 content495 - content-portion
    tray439 tray353 tray240 - tray
    table139 table188 table345 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray439 kitchen)
     (at tray353 kitchen)
     (at tray240 kitchen)
     (at_kitchen_bread bread288)
     (at_kitchen_bread bread173)
     (at_kitchen_bread bread193)
     (at_kitchen_content content54)
     (at_kitchen_content content66)
     (at_kitchen_content content495)
     (no_gluten_bread bread193)
     (no_gluten_content content54)
     (allergic_gluten child118)
     (not_allergic_gluten child50)
     (not_allergic_gluten child315)
     (waiting child315 table188)
     (waiting child50 table345)
     (waiting child118 table188)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child315)
     (served child50)
     (served child118)
    )
  )
)
