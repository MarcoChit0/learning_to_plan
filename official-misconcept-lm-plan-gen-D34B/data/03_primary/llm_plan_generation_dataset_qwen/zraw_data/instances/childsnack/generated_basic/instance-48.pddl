; child-snack task with 3 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 544554

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child402 child5 child440 - child
    bread4 bread177 bread194 - bread-portion
    content142 content16 content294 - content-portion
    tray273 tray94 tray346 - tray
    table37 table105 table471 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray273 kitchen)
     (at tray94 kitchen)
     (at tray346 kitchen)
     (at_kitchen_bread bread4)
     (at_kitchen_bread bread177)
     (at_kitchen_bread bread194)
     (at_kitchen_content content142)
     (at_kitchen_content content16)
     (at_kitchen_content content294)
     (no_gluten_bread bread194)
     (no_gluten_content content16)
     (allergic_gluten child440)
     (not_allergic_gluten child5)
     (not_allergic_gluten child402)
     (waiting child402 table105)
     (waiting child5 table105)
     (waiting child440 table105)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child402)
     (served child5)
     (served child440)
    )
  )
)
