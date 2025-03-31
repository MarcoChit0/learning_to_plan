; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 279381

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child93 child354 - child
    bread338 bread395 - bread-portion
    content168 content351 - content-portion
    tray329 - tray
    table254 table428 table383 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray329 kitchen)
     (at_kitchen_bread bread338)
     (at_kitchen_bread bread395)
     (at_kitchen_content content168)
     (at_kitchen_content content351)
     (no_gluten_bread bread395)
     (no_gluten_content content168)
     (allergic_gluten child354)
     (not_allergic_gluten child93)
     (waiting child93 table428)
     (waiting child354 table383)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child93)
     (served child354)
    )
  )
)
