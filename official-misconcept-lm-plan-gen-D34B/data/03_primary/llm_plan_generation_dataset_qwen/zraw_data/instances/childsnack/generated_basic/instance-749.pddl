; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 455075

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child50 child416 - child
    bread258 bread90 - bread-portion
    content190 content435 - content-portion
    tray399 tray463 - tray
    table348 table82 table84 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray399 kitchen)
     (at tray463 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_bread bread90)
     (at_kitchen_content content190)
     (at_kitchen_content content435)
     (no_gluten_bread bread90)
     (no_gluten_content content435)
     (allergic_gluten child50)
     (not_allergic_gluten child416)
     (waiting child50 table348)
     (waiting child416 table84)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child50)
     (served child416)
    )
  )
)
