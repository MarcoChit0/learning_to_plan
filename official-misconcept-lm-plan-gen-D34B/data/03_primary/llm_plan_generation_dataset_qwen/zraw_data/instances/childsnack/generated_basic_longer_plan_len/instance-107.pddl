; child-snack task with 4 children and 0.24 gluten factor 
; constant factor of 1.7
; random seed: 398605

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child225 child404 child246 child282 - child
    bread341 bread147 bread399 bread370 - bread-portion
    content82 content321 content289 content352 - content-portion
    tray228 tray175 tray332 - tray
    table51 table357 table198 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 sandw7 - sandwich
  )
  (:init
     (at tray228 kitchen)
     (at tray175 kitchen)
     (at tray332 kitchen)
     (at_kitchen_bread bread341)
     (at_kitchen_bread bread147)
     (at_kitchen_bread bread399)
     (at_kitchen_bread bread370)
     (at_kitchen_content content82)
     (at_kitchen_content content321)
     (at_kitchen_content content289)
     (at_kitchen_content content352)
     (not_allergic_gluten child246)
     (not_allergic_gluten child404)
     (not_allergic_gluten child282)
     (not_allergic_gluten child225)
     (waiting child225 table357)
     (waiting child404 table357)
     (waiting child246 table198)
     (waiting child282 table357)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
     (notexist sandw7)
  )
  (:goal
    (and
     (served child225)
     (served child404)
     (served child246)
     (served child282)
    )
  )
)
