; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 853045

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child407 child281 - child
    bread459 bread440 - bread-portion
    content201 content17 - content-portion
    tray157 tray79 tray394 - tray
    table151 table34 table102 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray157 kitchen)
     (at tray79 kitchen)
     (at tray394 kitchen)
     (at_kitchen_bread bread459)
     (at_kitchen_bread bread440)
     (at_kitchen_content content201)
     (at_kitchen_content content17)
     (no_gluten_bread bread459)
     (no_gluten_content content17)
     (allergic_gluten child281)
     (not_allergic_gluten child407)
     (waiting child407 table151)
     (waiting child281 table151)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child407)
     (served child281)
    )
  )
)
