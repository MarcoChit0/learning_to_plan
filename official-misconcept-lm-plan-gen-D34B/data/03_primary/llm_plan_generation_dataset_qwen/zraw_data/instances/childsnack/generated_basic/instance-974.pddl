; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 986362

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child19 child27 child43 - child
    bread122 bread357 bread349 - bread-portion
    content234 content250 content408 - content-portion
    tray100 tray29 tray144 tray335 - tray
    table38 table33 table291 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray100 kitchen)
     (at tray29 kitchen)
     (at tray144 kitchen)
     (at tray335 kitchen)
     (at_kitchen_bread bread122)
     (at_kitchen_bread bread357)
     (at_kitchen_bread bread349)
     (at_kitchen_content content234)
     (at_kitchen_content content250)
     (at_kitchen_content content408)
     (not_allergic_gluten child19)
     (not_allergic_gluten child27)
     (not_allergic_gluten child43)
     (waiting child19 table33)
     (waiting child27 table291)
     (waiting child43 table291)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child19)
     (served child27)
     (served child43)
    )
  )
)
