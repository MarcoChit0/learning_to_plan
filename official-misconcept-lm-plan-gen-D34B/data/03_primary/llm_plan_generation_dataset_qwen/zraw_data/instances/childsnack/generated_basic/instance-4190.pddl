; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 629188

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child155 child346 - child
    bread63 bread273 - bread-portion
    content110 content338 - content-portion
    tray488 - tray
    table392 table384 table492 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray488 kitchen)
     (at_kitchen_bread bread63)
     (at_kitchen_bread bread273)
     (at_kitchen_content content110)
     (at_kitchen_content content338)
     (not_allergic_gluten child346)
     (not_allergic_gluten child155)
     (waiting child155 table384)
     (waiting child346 table384)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child155)
     (served child346)
    )
  )
)
