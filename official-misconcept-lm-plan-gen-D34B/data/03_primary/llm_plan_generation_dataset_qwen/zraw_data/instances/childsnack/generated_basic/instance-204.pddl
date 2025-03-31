; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 738974

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child267 child273 - child
    bread238 bread278 - bread-portion
    content299 content278 - content-portion
    tray80 tray308 tray427 - tray
    table278 table446 table276 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray80 kitchen)
     (at tray308 kitchen)
     (at tray427 kitchen)
     (at_kitchen_bread bread238)
     (at_kitchen_bread bread278)
     (at_kitchen_content content299)
     (at_kitchen_content content278)
     (not_allergic_gluten child273)
     (not_allergic_gluten child267)
     (waiting child267 table446)
     (waiting child273 table446)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child267)
     (served child273)
    )
  )
)
