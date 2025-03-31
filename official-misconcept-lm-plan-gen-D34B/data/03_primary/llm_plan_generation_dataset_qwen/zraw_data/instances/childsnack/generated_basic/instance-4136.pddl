; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 859850

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child101 child374 - child
    bread220 bread177 - bread-portion
    content461 content126 - content-portion
    tray34 tray228 tray181 - tray
    table490 table342 table375 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray34 kitchen)
     (at tray228 kitchen)
     (at tray181 kitchen)
     (at_kitchen_bread bread220)
     (at_kitchen_bread bread177)
     (at_kitchen_content content461)
     (at_kitchen_content content126)
     (not_allergic_gluten child374)
     (not_allergic_gluten child101)
     (waiting child101 table490)
     (waiting child374 table375)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child101)
     (served child374)
    )
  )
)
