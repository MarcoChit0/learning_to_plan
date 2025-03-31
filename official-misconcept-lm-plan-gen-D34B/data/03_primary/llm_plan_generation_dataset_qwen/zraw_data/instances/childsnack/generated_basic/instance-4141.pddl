; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 63924

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child376 child402 - child
    bread24 bread114 - bread-portion
    content430 content435 - content-portion
    tray125 tray449 tray220 - tray
    table237 table380 table7 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray125 kitchen)
     (at tray449 kitchen)
     (at tray220 kitchen)
     (at_kitchen_bread bread24)
     (at_kitchen_bread bread114)
     (at_kitchen_content content430)
     (at_kitchen_content content435)
     (not_allergic_gluten child376)
     (not_allergic_gluten child402)
     (waiting child376 table237)
     (waiting child402 table7)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child376)
     (served child402)
    )
  )
)
