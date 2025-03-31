; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 946727

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child286 child403 - child
    bread200 bread103 - bread-portion
    content286 content339 - content-portion
    tray280 tray439 - tray
    table338 table231 table380 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray280 kitchen)
     (at tray439 kitchen)
     (at_kitchen_bread bread200)
     (at_kitchen_bread bread103)
     (at_kitchen_content content286)
     (at_kitchen_content content339)
     (not_allergic_gluten child403)
     (not_allergic_gluten child286)
     (waiting child286 table338)
     (waiting child403 table231)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child286)
     (served child403)
    )
  )
)
