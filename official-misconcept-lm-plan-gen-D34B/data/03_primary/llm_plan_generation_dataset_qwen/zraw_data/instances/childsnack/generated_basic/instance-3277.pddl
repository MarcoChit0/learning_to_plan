; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 363436

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child440 child129 - child
    bread349 bread208 - bread-portion
    content400 content212 - content-portion
    tray143 tray118 tray342 tray183 - tray
    table326 table17 table258 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray143 kitchen)
     (at tray118 kitchen)
     (at tray342 kitchen)
     (at tray183 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_bread bread208)
     (at_kitchen_content content400)
     (at_kitchen_content content212)
     (no_gluten_bread bread208)
     (no_gluten_content content212)
     (allergic_gluten child440)
     (not_allergic_gluten child129)
     (waiting child440 table258)
     (waiting child129 table326)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child440)
     (served child129)
    )
  )
)
