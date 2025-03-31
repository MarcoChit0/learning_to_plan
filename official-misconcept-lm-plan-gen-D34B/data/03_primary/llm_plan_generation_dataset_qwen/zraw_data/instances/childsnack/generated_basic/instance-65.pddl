; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 987434

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child11 child368 - child
    bread223 bread418 - bread-portion
    content146 content172 - content-portion
    tray311 tray299 - tray
    table99 table466 table325 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray311 kitchen)
     (at tray299 kitchen)
     (at_kitchen_bread bread223)
     (at_kitchen_bread bread418)
     (at_kitchen_content content146)
     (at_kitchen_content content172)
     (not_allergic_gluten child368)
     (not_allergic_gluten child11)
     (waiting child11 table99)
     (waiting child368 table99)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child11)
     (served child368)
    )
  )
)
