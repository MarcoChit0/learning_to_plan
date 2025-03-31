; child-snack task with 2 children and 0.56 gluten factor 
; constant factor of 1.3
; random seed: 904590

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child328 child326 - child
    bread434 bread215 - bread-portion
    content241 content316 - content-portion
    tray369 tray35 - tray
    table471 table431 table87 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray369 kitchen)
     (at tray35 kitchen)
     (at_kitchen_bread bread434)
     (at_kitchen_bread bread215)
     (at_kitchen_content content241)
     (at_kitchen_content content316)
     (no_gluten_bread bread434)
     (no_gluten_content content316)
     (allergic_gluten child326)
     (not_allergic_gluten child328)
     (waiting child328 table87)
     (waiting child326 table431)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child328)
     (served child326)
    )
  )
)
