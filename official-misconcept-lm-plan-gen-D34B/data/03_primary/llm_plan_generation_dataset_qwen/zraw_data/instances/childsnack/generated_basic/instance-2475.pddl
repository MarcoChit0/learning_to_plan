; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 769698

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child321 child407 - child
    bread234 bread148 - bread-portion
    content219 content72 - content-portion
    tray36 tray170 - tray
    table277 table58 table226 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray36 kitchen)
     (at tray170 kitchen)
     (at_kitchen_bread bread234)
     (at_kitchen_bread bread148)
     (at_kitchen_content content219)
     (at_kitchen_content content72)
     (not_allergic_gluten child321)
     (not_allergic_gluten child407)
     (waiting child321 table58)
     (waiting child407 table58)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child321)
     (served child407)
    )
  )
)
