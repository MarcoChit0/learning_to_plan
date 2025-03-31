; child-snack task with 4 children and 0.24 gluten factor 
; constant factor of 1.7
; random seed: 160913

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child56 child77 child321 child475 - child
    bread351 bread64 bread440 bread158 - bread-portion
    content420 content163 content22 content162 - content-portion
    tray69 tray351 tray392 - tray
    table358 table209 table243 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 sandw7 - sandwich
  )
  (:init
     (at tray69 kitchen)
     (at tray351 kitchen)
     (at tray392 kitchen)
     (at_kitchen_bread bread351)
     (at_kitchen_bread bread64)
     (at_kitchen_bread bread440)
     (at_kitchen_bread bread158)
     (at_kitchen_content content420)
     (at_kitchen_content content163)
     (at_kitchen_content content22)
     (at_kitchen_content content162)
     (not_allergic_gluten child475)
     (not_allergic_gluten child77)
     (not_allergic_gluten child56)
     (not_allergic_gluten child321)
     (waiting child56 table209)
     (waiting child77 table243)
     (waiting child321 table243)
     (waiting child475 table209)
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
     (served child56)
     (served child77)
     (served child321)
     (served child475)
    )
  )
)
