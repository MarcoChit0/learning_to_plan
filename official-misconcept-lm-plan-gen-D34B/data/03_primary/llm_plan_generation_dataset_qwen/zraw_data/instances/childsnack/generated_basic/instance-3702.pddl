; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 387067

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child372 child91 - child
    bread64 bread451 - bread-portion
    content333 content131 - content-portion
    tray171 tray183 - tray
    table38 table233 table244 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray171 kitchen)
     (at tray183 kitchen)
     (at_kitchen_bread bread64)
     (at_kitchen_bread bread451)
     (at_kitchen_content content333)
     (at_kitchen_content content131)
     (no_gluten_bread bread451)
     (no_gluten_content content131)
     (allergic_gluten child91)
     (not_allergic_gluten child372)
     (waiting child372 table233)
     (waiting child91 table233)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child372)
     (served child91)
    )
  )
)
