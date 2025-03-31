; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 691328

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child382 child459 - child
    bread382 bread180 - bread-portion
    content41 content493 - content-portion
    tray348 - tray
    table340 table364 table265 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray348 kitchen)
     (at_kitchen_bread bread382)
     (at_kitchen_bread bread180)
     (at_kitchen_content content41)
     (at_kitchen_content content493)
     (no_gluten_bread bread382)
     (no_gluten_content content41)
     (allergic_gluten child382)
     (not_allergic_gluten child459)
     (waiting child382 table340)
     (waiting child459 table340)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child382)
     (served child459)
    )
  )
)
