; child-snack task with 3 children and 0.7 gluten factor 
; constant factor of 1.7
; random seed: 665218

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child404 child6 child403 - child
    bread333 bread73 bread499 - bread-portion
    content416 content76 content45 - content-portion
    tray218 tray428 - tray
    table482 table20 table417 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray218 kitchen)
     (at tray428 kitchen)
     (at_kitchen_bread bread333)
     (at_kitchen_bread bread73)
     (at_kitchen_bread bread499)
     (at_kitchen_content content416)
     (at_kitchen_content content76)
     (at_kitchen_content content45)
     (no_gluten_bread bread333)
     (no_gluten_bread bread73)
     (no_gluten_content content76)
     (no_gluten_content content45)
     (allergic_gluten child6)
     (allergic_gluten child403)
     (not_allergic_gluten child404)
     (waiting child404 table482)
     (waiting child6 table417)
     (waiting child403 table482)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child404)
     (served child6)
     (served child403)
    )
  )
)
