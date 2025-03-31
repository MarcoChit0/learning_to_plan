; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 511539

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child195 child40 - child
    bread258 bread41 - bread-portion
    content168 content199 - content-portion
    tray185 tray84 - tray
    table22 table132 table274 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray185 kitchen)
     (at tray84 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_bread bread41)
     (at_kitchen_content content168)
     (at_kitchen_content content199)
     (no_gluten_bread bread258)
     (no_gluten_content content168)
     (allergic_gluten child195)
     (not_allergic_gluten child40)
     (waiting child195 table274)
     (waiting child40 table22)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child195)
     (served child40)
    )
  )
)
