; child-snack task with 3 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 594693

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child383 child231 child359 - child
    bread395 bread254 bread299 - bread-portion
    content270 content176 content168 - content-portion
    tray172 tray169 tray348 tray27 - tray
    table393 table333 table99 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray172 kitchen)
     (at tray169 kitchen)
     (at tray348 kitchen)
     (at tray27 kitchen)
     (at_kitchen_bread bread395)
     (at_kitchen_bread bread254)
     (at_kitchen_bread bread299)
     (at_kitchen_content content270)
     (at_kitchen_content content176)
     (at_kitchen_content content168)
     (no_gluten_bread bread299)
     (no_gluten_content content168)
     (allergic_gluten child359)
     (not_allergic_gluten child231)
     (not_allergic_gluten child383)
     (waiting child383 table393)
     (waiting child231 table99)
     (waiting child359 table393)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child383)
     (served child231)
     (served child359)
    )
  )
)
