; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 142863

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child256 child35 - child
    bread199 bread205 - bread-portion
    content139 content279 - content-portion
    tray57 tray369 - tray
    table234 table382 table249 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray57 kitchen)
     (at tray369 kitchen)
     (at_kitchen_bread bread199)
     (at_kitchen_bread bread205)
     (at_kitchen_content content139)
     (at_kitchen_content content279)
     (not_allergic_gluten child35)
     (not_allergic_gluten child256)
     (waiting child256 table234)
     (waiting child35 table382)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child256)
     (served child35)
    )
  )
)
