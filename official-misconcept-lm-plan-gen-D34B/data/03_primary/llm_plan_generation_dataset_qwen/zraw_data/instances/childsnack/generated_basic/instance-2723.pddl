; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 483811

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child334 child40 child153 - child
    bread276 bread103 bread361 - bread-portion
    content78 content105 content156 - content-portion
    tray242 tray33 tray84 - tray
    table198 table71 table359 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray242 kitchen)
     (at tray33 kitchen)
     (at tray84 kitchen)
     (at_kitchen_bread bread276)
     (at_kitchen_bread bread103)
     (at_kitchen_bread bread361)
     (at_kitchen_content content78)
     (at_kitchen_content content105)
     (at_kitchen_content content156)
     (not_allergic_gluten child40)
     (not_allergic_gluten child153)
     (not_allergic_gluten child334)
     (waiting child334 table198)
     (waiting child40 table359)
     (waiting child153 table71)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child334)
     (served child40)
     (served child153)
    )
  )
)
